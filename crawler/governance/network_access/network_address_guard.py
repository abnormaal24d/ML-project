"""Network address guard for crawler HTTP targets (SSRF protection etc).

This module owns the rules decision only. It does not perform HTTP requests or
DNS lookups. URL filters and redirect checks can use it to reject literal
private/local targets early; the resolver adapter applies the same rules after
DNS resolution.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable, Mapping
from typing import TypeVar
from urllib.parse import urlparse

from config.collection.http_rules import NetworkAccessSettings
from crawler.governance.domains.host_normalizer import HostNormalizer
from logger.project_logger import ProjectLogger

_SHARED_ADDRESS_SPACE = ipaddress.ip_network("100.64.0.0/10")
_CLOUD_METADATA_HOSTS = frozenset(
    {
        "instance-data",
        "metadata.azure.internal",
        "metadata.google.internal",
    }
)
IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
AddressRecord = TypeVar("AddressRecord", bound=Mapping[str, object])


class NetworkAddressGuard:
    """Reject URLs and resolved addresses targeting private infrastructure."""

    def __init__(
        self,
        *,
        settings: NetworkAccessSettings,
        host_normalizer: HostNormalizer,
        logger: ProjectLogger,
    ) -> None:
        self._settings = settings
        self._host_normalizer = host_normalizer
        self._logger = logger
        self._allowed_ip_literals = self._normalize_allowed_ip_literals(
            values=settings.allowed_ip_literals
        )
        self._blocked_hostname_suffixes = tuple(
            suffix
            for suffix in (
                self._normalize_hostname_suffix(value)
                for value in settings.blocked_hostname_suffixes
            )
            if suffix
        )

    @property
    def dns_resolution_enabled(self) -> bool:
        """Return whether resolved-address checks should wrap DNS lookups."""

        return self._settings.enforce_on_dns_resolution

    @property
    def host_normalizer(self) -> HostNormalizer:
        """Return the shared canonicalizer used by downstream DNS state."""

        return self._host_normalizer

    # ------------------------------------------------------------------
    # URL and host rejection rules
    # ------------------------------------------------------------------
    def rejection_reason_for_url(self, url: str) -> str | None:
        """Return a rejection reason for literal/local URL hosts, if any."""

        try:
            parsed = urlparse(url)
        except ValueError:
            return "invalid_url"

        try:
            host = parsed.hostname
        except ValueError:
            return "invalid_url"

        if self._settings.block_url_credentials and (
            parsed.username is not None or parsed.password is not None
        ):
            return "url_credentials_blocked"

        try:
            port = parsed.port
        except ValueError:
            return "invalid_url"

        scheme = parsed.scheme.lower()

        if port is not None:
            if (
                scheme == "http"
                and port not in self._settings.allowed_http_ports
            ):
                return "network_port_blocked"

            if (
                scheme == "https"
                and port not in self._settings.allowed_https_ports
            ):
                return "network_port_blocked"

        return self.rejection_reason_for_host(host or "")

    def rejection_reason_for_host(self, host: str) -> str | None:
        """
        Return a rejection reason for a hostname or IP literal, if blocked.
        """

        normalized_host = self._normalize_host(host)
        if not normalized_host:
            raw_address = self._parse_ip_address(host)
            if raw_address is not None:
                return self.rejection_reason_for_address(str(raw_address))
            return "invalid_hostname"

        if normalized_host in _CLOUD_METADATA_HOSTS:
            return "cloud_metadata_hostname_blocked"

        if self._local_hostname_blocked(host=normalized_host):
            return "local_hostname_blocked"

        address = self._parse_ip_address(normalized_host)
        if address is None:
            return None

        return self.rejection_reason_for_address(str(address))

    # ------------------------------------------------------------------
    # Address level rules
    # ------------------------------------------------------------------
    def rejection_reason_for_address(self, address: str) -> str | None:
        """Return why a resolved IP address is unsafe, or None when allowed."""

        ip_address = self._parse_ip_address(address)
        if ip_address is None:
            return "invalid_ip_address"

        normalized = self._canonical_address(ip_address)
        if normalized in self._allowed_ip_literals:
            return None

        if (
            self._settings.block_unspecified_ip_ranges
            and normalized.is_unspecified
        ):
            return "unspecified_ip_blocked"

        if self._settings.block_loopback_ip_ranges and normalized.is_loopback:
            return "loopback_ip_blocked"

        if (
            self._settings.block_link_local_ip_ranges
            and normalized.is_link_local
        ):
            return "link_local_ip_blocked"

        if (
            self._settings.block_private_ip_ranges
            and self._is_private_or_shared_address(normalized)
        ):
            return "private_ip_blocked"

        if (
            self._settings.block_multicast_ip_ranges
            and normalized.is_multicast
        ):
            return "multicast_ip_blocked"

        if self._settings.block_reserved_ip_ranges and normalized.is_reserved:
            return "reserved_ip_blocked"

        if self._settings.block_site_local_ip_ranges and getattr(
            normalized, "is_site_local", False
        ):
            return "site_local_ip_blocked"

        return None

    # ------------------------------------------------------------------
    # DNS resolution filtering
    # ------------------------------------------------------------------
    def filter_resolved_addresses(
        self,
        *,
        host: str,
        addresses: list[AddressRecord],
    ) -> list[AddressRecord]:
        """Reject an entire DNS answer set when any address is unsafe."""

        allowed: list[AddressRecord] = []
        blocked = False
        for item in addresses:
            address = str(item.get("host") or "")
            reason = self.rejection_reason_for_address(address)
            if reason is None:
                allowed.append(item)
                continue

            self._logger.warning(
                "network_access_dns_address_rejected",
                extra={
                    "host": host,
                    "address": address,
                    "reason": reason,
                },
            )
            blocked = True

        return [] if blocked else allowed

    def _local_hostname_blocked(self, *, host: str) -> bool:
        if not self._settings.block_local_hostnames:
            return False
        return any(
            host == suffix.lstrip(".") or host.endswith(suffix)
            for suffix in self._blocked_hostname_suffixes
        )

    @classmethod
    def _parse_ip_address(cls, value: str) -> IPAddress | None:
        normalized_value = value.strip("[]")
        try:
            return ipaddress.ip_address(normalized_value)
        except ValueError:
            return cls._parse_obfuscated_ipv4_address(normalized_value)

    @classmethod
    def _parse_obfuscated_ipv4_address(
        cls,
        value: str,
    ) -> ipaddress.IPv4Address | None:
        """Parse noncanonical IPv4 spellings that URL parsers may accept.

        Some HTTP stacks historically accepted integer, octal, hex, and
        shortened dotted IPv4 forms such as ``2130706433`` or
        ``0177.0.0.1``. Treating these as ordinary hostnames creates an SSRF
        gap, because they can still route to loopback/private networks.
        """

        if not value or ":" in value:
            return None

        parts = value.split(".")
        if len(parts) > 4 or any(part == "" for part in parts):
            return None

        numbers = tuple(cls._parse_ipv4_number(part) for part in parts)
        if any(number is None for number in numbers):
            return None
        octets = tuple(number for number in numbers if number is not None)

        try:
            if len(octets) == 1:
                return cls._ipv4_from_integer(octets[0])
            if len(octets) == 2:
                first, rest = octets
                if first > 0xFF or rest > 0xFFFFFF:
                    return None
                return cls._ipv4_from_integer((first << 24) | rest)
            if len(octets) == 3:
                first, second, rest = octets
                if first > 0xFF or second > 0xFF or rest > 0xFFFF:
                    return None
                return cls._ipv4_from_integer(
                    (first << 24) | (second << 16) | rest
                )
            if any(octet > 0xFF for octet in octets):
                return None
            first, second, third, fourth = octets
            return cls._ipv4_from_integer(
                (first << 24) | (second << 16) | (third << 8) | fourth
            )
        except ValueError:
            return None

    @staticmethod
    def _parse_ipv4_number(part: str) -> int | None:
        base = 10
        digits = part
        if part.lower().startswith("0x"):
            base = 16
            digits = part[2:]
        elif len(part) > 1 and part.startswith("0"):
            base = 8
            digits = part[1:] or "0"
        if not digits:
            return None
        try:
            number = int(digits, base)
        except ValueError:
            return None
        return number if 0 <= number <= 0xFFFFFFFF else None

    @staticmethod
    def _ipv4_from_integer(value: int) -> ipaddress.IPv4Address:
        if not 0 <= value <= 0xFFFFFFFF:
            raise ValueError("IPv4 integer outside valid range")
        return ipaddress.IPv4Address(value)

    @staticmethod
    def _canonical_address(address: IPAddress) -> IPAddress:
        mapped = getattr(address, "ipv4_mapped", None)
        return mapped or address

    @classmethod
    def _normalize_address(cls, value: object) -> str:
        address = cls._parse_ip_address(str(value))
        if address is None:
            return ""
        return str(cls._canonical_address(address))

    def _normalize_allowed_ip_literals(
        self,
        *,
        values: object,
    ) -> set[str]:
        normalized_values: set[str] = set()
        if values is None:
            candidates: Iterable[object] = ()
        elif isinstance(values, str):
            candidates = (values,)
        elif isinstance(values, Iterable):
            candidates = values
        else:
            candidates = (values,)

        for value in candidates:
            normalized = self._normalize_address(value)
            if normalized:
                normalized_values.add(normalized)
                continue
            self._logger.warning(
                "network_access_invalid_allowed_ip_literal",
                extra={"value": str(value)},
            )
        return normalized_values

    def _normalize_host(self, host: str) -> str:
        normalized = self._host_normalizer.normalize(host)
        return normalized or ""

    def _normalize_hostname_suffix(self, value: object) -> str:
        if not isinstance(value, str):
            return ""
        suffix = self._host_normalizer.normalize(value.strip().lstrip("."))
        if suffix is None:
            return ""
        return suffix if suffix == "localhost" else f".{suffix}"

    @staticmethod
    def _is_private_or_shared_address(address: IPAddress) -> bool:
        if address.version == 4 and address in _SHARED_ADDRESS_SPACE:
            return True
        return bool(address.is_private)
