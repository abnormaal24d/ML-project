"""Contract tests for captured aiohttp connected-peer extraction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from crawler.fetching.network.session import connected_peer_address


class _Transport:
    def __init__(self, peer: object) -> None:
        self._peer = peer

    def get_extra_info(self, name: str) -> object:
        assert name == "peername"
        return self._peer


class _Response:
    def __init__(
        self,
        *,
        connected_peer: object | None = None,
        connection: object | None = None,
        protocol: object | None = None,
    ) -> None:
        self.connected_peer = connected_peer
        self.connection = connection
        self._protocol = protocol


def test_connected_peer_address_reads_captured_response_metadata() -> None:
    response = _Response(
        connected_peer=("93.184.216.34", 443),
        connection=SimpleNamespace(
            transport=_Transport(("198.51.100.10", 443))
        ),
    )

    assert connected_peer_address(response) == "93.184.216.34"  # type: ignore[arg-type]


def test_connected_peer_address_does_not_inspect_connection_or_protocol() -> (
    None
):
    connection_response = _Response(
        connection=SimpleNamespace(transport=_Transport(("203.0.113.1", 443)))
    )
    protocol_response = _Response(
        connection=SimpleNamespace(transport=_Transport(None)),
        protocol=SimpleNamespace(transport=_Transport(("203.0.113.2", 443))),
    )

    assert connected_peer_address(connection_response) is None  # type: ignore[arg-type]
    assert connected_peer_address(protocol_response) is None  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "response",
    (
        _Response(),
        _Response(connected_peer=()),
        _Response(connected_peer="203.0.113.3"),
        _Response(connection=SimpleNamespace(transport=object())),
    ),
)
def test_connected_peer_address_rejects_unavailable_or_invalid_metadata(
    response: _Response,
) -> None:
    assert connected_peer_address(response) is None  # type: ignore[arg-type]
