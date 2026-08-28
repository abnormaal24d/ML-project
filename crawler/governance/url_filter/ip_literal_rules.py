"""Apply rejection rules to blocked or unsafe IP-literal hosts."""

from __future__ import annotations

import ipaddress

_CARRIER_GRADE_NAT = ipaddress.ip_network("100.64.0.0/10")


class IpLiteralRules:
    def __init__(self, *, blocked_ip_literals: set[str]) -> None:
        self._blocked_ip_literals = blocked_ip_literals

    def rejection_reason(self, host: str) -> str | None:
        if not host:
            return None
        try:
            address = ipaddress.ip_address(host.strip("[]"))
        except ValueError:
            return None
        if str(address) in self._blocked_ip_literals:
            return "configured_blocked_host"
        if self._is_unsafe_address(address):
            return "unsafe_ip_literal"
        return None

    @staticmethod
    def _is_unsafe_address(
        address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    ) -> bool:
        return (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
            or getattr(address, "is_site_local", False)
            or (address.version == 4 and address in _CARRIER_GRADE_NAT)
        )
