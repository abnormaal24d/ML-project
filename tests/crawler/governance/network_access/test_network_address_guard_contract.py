from __future__ import annotations

import pytest

from config.collection.http_rules import NetworkAccessSettings
from crawler.governance.domains.host_normalizer import HostNormalizer
from crawler.governance.network_access.network_address_guard import (
    NetworkAddressGuard,
)
from crawler.governance.network_access.safe_urllib import SafeUrlOpener
from crawler.governance.url_filter.ip_literal_rules import IpLiteralRules


class _CapturingLogger:
    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict[str, object]]] = []

    def warning(self, event: str, **fields: object) -> None:
        self.warnings.append((event, fields))

    def debug(self, event: str, **fields: object) -> None:
        del event, fields

    def info(self, event: str, **fields: object) -> None:
        del event, fields

    def error(self, event: str, **fields: object) -> None:
        del event, fields


def _logger() -> _CapturingLogger:
    return _CapturingLogger()


def _guard(
    settings: NetworkAccessSettings | None = None,
    logger: _CapturingLogger | None = None,
) -> NetworkAddressGuard:
    return NetworkAddressGuard(
        settings=(
            settings if settings is not None else NetworkAccessSettings()
        ),
        host_normalizer=HostNormalizer(),
        logger=logger if logger is not None else _logger(),
    )


def _opener(guard: NetworkAddressGuard) -> SafeUrlOpener:
    return SafeUrlOpener(network_access_guard=guard)


# --- loopback / link-local / private / shared block reasons ---------------


def test_loopback_ipv4_is_blocked() -> None:
    assert _guard().rejection_reason_for_address("127.0.0.1") == (
        "loopback_ip_blocked"
    )
    assert _guard().rejection_reason_for_host("127.0.0.1") == (
        "loopback_ip_blocked"
    )


def test_loopback_ipv6_is_blocked() -> None:
    assert (
        _guard().rejection_reason_for_address("::1") == "loopback_ip_blocked"
    )
    assert _guard().rejection_reason_for_host("[::1]") == (
        "loopback_ip_blocked"
    )


def test_ipv4_mapped_loopback_is_blocked() -> None:
    assert _guard().rejection_reason_for_address("::ffff:127.0.0.1") == (
        "loopback_ip_blocked"
    )


def test_link_local_ipv4_is_blocked() -> None:
    guard = _guard()
    assert guard.rejection_reason_for_address("169.254.0.1") == (
        "link_local_ip_blocked"
    )
    assert guard.rejection_reason_for_address("169.254.169.254") == (
        "link_local_ip_blocked"
    )


def test_link_local_ipv6_is_blocked() -> None:
    assert _guard().rejection_reason_for_address("fe80::1") == (
        "link_local_ip_blocked"
    )


def test_private_ipv4_ranges_are_blocked() -> None:
    guard = _guard()
    for address in (
        "10.0.0.1",
        "10.255.255.255",
        "172.16.0.1",
        "172.31.255.255",
        "192.168.1.1",
    ):
        assert guard.rejection_reason_for_address(address) == (
            "private_ip_blocked"
        ), address


def test_shared_cgnat_space_is_blocked_as_private() -> None:
    guard = _guard()
    assert guard.rejection_reason_for_address("100.64.0.0") == (
        "private_ip_blocked"
    )
    assert guard.rejection_reason_for_address("100.64.0.1") == (
        "private_ip_blocked"
    )
    assert guard.rejection_reason_for_address("100.127.255.255") == (
        "private_ip_blocked"
    )
    assert guard.rejection_reason_for_host("100.65.1.1") == (
        "private_ip_blocked"
    )


def test_shared_address_space_does_not_leak_to_borders() -> None:
    guard = _guard()
    assert guard.rejection_reason_for_address("100.63.255.255") is None
    assert guard.rejection_reason_for_address("100.128.0.1") is None


def test_private_ipv6_is_blocked() -> None:
    guard = _guard()
    assert guard.rejection_reason_for_address("fd00::1") == (
        "private_ip_blocked"
    )
    assert guard.rejection_reason_for_address("2001:db8::1") == (
        "private_ip_blocked"
    )


def test_unspecified_ranges_are_blocked() -> None:
    guard = _guard()
    assert guard.rejection_reason_for_address("0.0.0.0") == (
        "unspecified_ip_blocked"
    )
    assert guard.rejection_reason_for_address("::") == "unspecified_ip_blocked"


def test_multicast_ranges_are_blocked() -> None:
    guard = _guard()
    assert guard.rejection_reason_for_address("224.0.0.1") == (
        "multicast_ip_blocked"
    )
    assert guard.rejection_reason_for_address("ff02::1") == (
        "multicast_ip_blocked"
    )


def test_reserved_ipv4_hits_private_reason_by_default() -> None:
    assert _guard().rejection_reason_for_address("240.0.0.1") == (
        "private_ip_blocked"
    )


def test_reserved_range_hits_reserved_reason_when_private_unblocked() -> None:
    guard = _guard(
        NetworkAccessSettings(block_private_ip_ranges=False),
    )
    assert guard.rejection_reason_for_address("255.255.255.255") == (
        "reserved_ip_blocked"
    )


def test_site_local_ipv6_is_blocked() -> None:
    assert _guard().rejection_reason_for_address("fec0::1") == (
        "site_local_ip_blocked"
    )


# --- public ranges and hostnames are allowed -------------------------------


def test_public_ipv4_addresses_are_allowed() -> None:
    guard = _guard()
    for address in ("8.8.8.8", "93.184.216.34", "100.63.255.255", "1.2.3.4"):
        assert guard.rejection_reason_for_address(address) is None, address


def test_public_ipv6_addresses_are_allowed() -> None:
    assert (
        _guard().rejection_reason_for_address("2606:4700:4700::1111") is None
    )


def test_public_hostnames_are_allowed() -> None:
    guard = _guard()
    assert guard.rejection_reason_for_host("example.com") is None
    assert guard.rejection_reason_for_host("sub.example.com") is None
    assert guard.rejection_reason_for_url("https://example.com/path") is None


def test_invalid_addresses_get_invalid_reasons() -> None:
    guard = _guard()
    assert guard.rejection_reason_for_address("not-an-ip") == (
        "invalid_ip_address"
    )
    assert guard.rejection_reason_for_host("") == "invalid_hostname"
    assert guard.rejection_reason_for_host("bad host") == "invalid_hostname"


def test_malformed_ipv4_is_rejected() -> None:
    guard = _guard()
    assert guard.rejection_reason_for_host("999.1.1.1") == "invalid_hostname"
    assert guard.rejection_reason_for_host("1.2.3.4.5") == "invalid_hostname"
    assert guard.rejection_reason_for_host("999999999999999") == (
        "invalid_hostname"
    )


# --- obfuscated / integer-encoded IPv4 ------------------------------------


def test_integer_encoded_ipv4_loopback_is_blocked() -> None:
    assert _guard().rejection_reason_for_host("2130706433") == (
        "loopback_ip_blocked"
    )


def test_integer_encoded_ipv4_private_is_blocked() -> None:
    guard = _guard()
    assert guard.rejection_reason_for_host("3232235777") == (
        "private_ip_blocked"
    )
    assert guard.rejection_reason_for_host("167772161") == (
        "private_ip_blocked"
    )
    assert guard.rejection_reason_for_host("2886729728") == (
        "private_ip_blocked"
    )


def test_octal_hex_and_shortened_ipv4_are_blocked() -> None:
    guard = _guard()
    assert guard.rejection_reason_for_host("0177.0.0.1") == (
        "loopback_ip_blocked"
    )
    assert guard.rejection_reason_for_host("0x7f.1") == "loopback_ip_blocked"
    assert guard.rejection_reason_for_host("0x7f000001") == (
        "loopback_ip_blocked"
    )
    assert guard.rejection_reason_for_host("127.1") == "loopback_ip_blocked"
    assert guard.rejection_reason_for_host("127.0.1") == "loopback_ip_blocked"


def test_integer_encoded_ipv4_in_url_is_blocked() -> None:
    assert _guard().rejection_reason_for_url("http://2130706433/path") == (
        "loopback_ip_blocked"
    )


def test_bracketed_ipv6_literal_url_is_blocked() -> None:
    assert _guard().rejection_reason_for_url("http://[::1]/path") == (
        "loopback_ip_blocked"
    )


# --- settings variations ---------------------------------------------------


def test_loopback_block_toggle_falls_back_to_private() -> None:
    guard = _guard(
        NetworkAccessSettings(block_loopback_ip_ranges=False),
    )
    assert guard.rejection_reason_for_address("127.0.0.1") == (
        "private_ip_blocked"
    )


def test_all_ip_block_flags_can_be_toggled_off() -> None:
    guard = _guard(
        NetworkAccessSettings(
            block_private_ip_ranges=False,
            block_loopback_ip_ranges=False,
            block_link_local_ip_ranges=False,
            block_multicast_ip_ranges=False,
            block_reserved_ip_ranges=False,
            block_unspecified_ip_ranges=False,
            block_site_local_ip_ranges=False,
        ),
    )
    assert guard.rejection_reason_for_address("127.0.0.1") is None
    assert guard.rejection_reason_for_address("10.0.0.1") is None


def test_blocked_hostname_suffixes_guard_local_names() -> None:
    guard = _guard()
    for host in (
        "localhost",
        "localhost.localdomain",
        "foo.local",
        "dns.localdomain",
        "ip6-localhost",
    ):
        assert guard.rejection_reason_for_host(host) == (
            "local_hostname_blocked"
        ), host
    assert guard.rejection_reason_for_host("localhost.example.com") is None


def test_local_hostname_block_can_be_disabled() -> None:
    guard = _guard(
        NetworkAccessSettings(block_local_hostnames=False),
    )
    assert guard.rejection_reason_for_host("localhost") is None


def test_cloud_metadata_hostnames_are_blocked() -> None:
    guard = _guard()
    for host in (
        "instance-data",
        "metadata.azure.internal",
        "metadata.google.internal",
    ):
        assert guard.rejection_reason_for_host(host) == (
            "cloud_metadata_hostname_blocked"
        ), host


def test_url_credentials_are_blocked() -> None:
    assert (
        _guard().rejection_reason_for_url("http://user:pass@example.com/")
        == "url_credentials_blocked"
    )


def test_default_http_port_is_allowed() -> None:
    guard = _guard()

    assert guard.rejection_reason_for_url("http://example.com:80/path") is None


def test_default_https_port_is_allowed() -> None:
    guard = _guard()

    assert (
        guard.rejection_reason_for_url("https://example.com:443/path") is None
    )


@pytest.mark.parametrize(
    "url",
    (
        "http://example.com:81/path",
        "http://example.com:8080/path",
        "https://example.com:8443/path",
        "https://example.com:2375/path",
        "https://example.com:6379/path",
    ),
)
def test_non_allowed_network_ports_are_blocked(
    url: str,
) -> None:
    assert _guard().rejection_reason_for_url(url) == "network_port_blocked"


def test_configured_https_port_is_allowed() -> None:
    guard = _guard(
        NetworkAccessSettings(
            allowed_https_ports=(443, 8443),
        )
    )

    assert (
        guard.rejection_reason_for_url("https://example.com:8443/path") is None
    )


def test_configured_http_port_is_allowed() -> None:
    guard = _guard(
        NetworkAccessSettings(
            allowed_http_ports=(80, 8080),
        )
    )

    assert (
        guard.rejection_reason_for_url("http://example.com:8080/path") is None
    )


def test_implicit_ports_are_allowed() -> None:
    guard = _guard()

    # Implicit ports (no explicit port in URL) should be allowed
    assert guard.rejection_reason_for_url("http://example.com/path") is None
    assert guard.rejection_reason_for_url("https://example.com/path") is None


def test_url_credentials_block_can_be_disabled() -> None:
    guard = _guard(
        NetworkAccessSettings(block_url_credentials=False),
    )
    assert guard.rejection_reason_for_url("http://user:pass@8.8.8.8/") is None


def test_configured_allowed_literals_are_canonicalized_but_not_bypassing() -> (
    None
):
    guard = _guard(
        NetworkAccessSettings(allowed_ip_literals=("127.0.0.1",)),
    )
    assert guard.rejection_reason_for_address("2130706433") == (
        "loopback_ip_blocked"
    )


# --- DNS resolution answer-set filtering -----------------------------------


def test_filter_all_public_answer_set_is_returned() -> None:
    logger = _logger()
    guard = _guard(logger=logger)
    items = [{"host": "8.8.8.8"}, {"host": "93.184.216.34"}]
    assert (
        guard.filter_resolved_addresses(host="public.test", addresses=items)
        == items
    )
    assert logger.warnings == []


def test_filter_any_blocked_address_voids_the_answer_set() -> None:
    logger = _logger()
    guard = _guard(logger=logger)
    items = [{"host": "8.8.8.8"}, {"host": "10.0.0.1"}]
    assert (
        guard.filter_resolved_addresses(host="mixed.test", addresses=items)
        == []
    )
    assert [event for event, _ in logger.warnings] == [
        "network_access_dns_address_rejected"
    ]


def test_filter_empty_answer_set_is_returned() -> None:
    logger = _logger()
    guard = _guard(logger=logger)
    assert (
        guard.filter_resolved_addresses(host="none.test", addresses=[]) == []
    )
    assert logger.warnings == []


# --- safe_urllib pre-network rejection (no DNS, no sockets) ----------------


def test_opener_rejects_loopback_before_dns() -> None:
    with pytest.raises(
        RuntimeError,
        match="blocked_network_target:loopback_ip_blocked",
    ):
        _opener(_guard()).validate_url(url="http://127.0.0.1/x")


def test_opener_rejects_integer_encoded_ipv4_before_dns() -> None:
    with pytest.raises(
        RuntimeError,
        match="blocked_network_target:loopback_ip_blocked",
    ):
        _opener(_guard()).validate_url(url="http://2130706433/")


def test_opener_rejects_metadata_link_local_before_dns() -> None:
    with pytest.raises(
        RuntimeError,
        match="blocked_network_target:link_local_ip_blocked",
    ):
        _opener(_guard()).validate_url(
            url="http://169.254.169.254/latest/meta-data/"
        )


def test_opener_rejects_cloud_metadata_hostname_before_dns() -> None:
    with pytest.raises(
        RuntimeError,
        match="blocked_network_target:cloud_metadata_hostname_blocked",
    ):
        _opener(_guard()).validate_url(url="http://metadata.google.internal/")


def test_opener_rejects_unsupported_scheme() -> None:
    with pytest.raises(RuntimeError, match="unsupported_network_url"):
        _opener(_guard()).validate_url(url="ftp://8.8.8.8/x")


def test_opener_rejects_invalid_port_before_network() -> None:
    with pytest.raises(RuntimeError, match="invalid_network_url"):
        _opener(_guard()).validate_url(url="http://8.8.8.8:99999/x")


# --- IP literal host rules --------------------------------------------------


def _literal_rules() -> IpLiteralRules:
    return IpLiteralRules(blocked_ip_literals={"1.2.3.4"})


def test_ip_literal_configured_blocked_host_wins() -> None:
    assert _literal_rules().rejection_reason("1.2.3.4") == (
        "configured_blocked_host"
    )


def test_ip_literal_unsafe_ranges_are_rejected() -> None:
    rules = _literal_rules()
    for host in (
        "127.0.0.1",
        "::1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.0.1",
        "fe80::1",
        "100.64.0.1",
        "224.0.0.1",
        "240.0.0.1",
        "0.0.0.0",
        "fec0::1",
        "[::1]",
    ):
        assert rules.rejection_reason(host) == "unsafe_ip_literal", host


def test_ip_literal_public_and_hostnames_are_allowed() -> None:
    rules = _literal_rules()
    assert rules.rejection_reason("8.8.8.8") is None
    assert rules.rejection_reason("2606:4700:4700::1111") is None
    assert rules.rejection_reason("example.com") is None
    assert rules.rejection_reason("") is None
