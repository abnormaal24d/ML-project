"""Bounded video probe downloaders for metadata fallbacks."""

from __future__ import annotations

import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config.environment.default_values import DEFAULT_VIDEO_METADATA_USER_AGENT
from crawler.governance.network_access.safe_urllib import SafeUrlOpener

if TYPE_CHECKING:
    from crawler.governance.network_access.network_address_guard import (
        NetworkAddressGuard,
    )
    from crawler.governance.redirect.redirect_rules_validator import (
        RedirectRulesValidator,
    )


class UrlopenTransport:
    """Synchronous urlopen transport for bounded metadata probes."""

    def __init__(
        self,
        *,
        network_access_guard: NetworkAddressGuard,
        redirector: RedirectRulesValidator,
        max_redirects: int,
        source_name: str | None = None,
    ) -> None:
        self._url_opener = SafeUrlOpener(
            network_access_guard=network_access_guard,
            redirector=redirector,
            max_redirects=max_redirects,
            source_name=source_name,
        )

    def open(
        self,
        request: urllib.request.Request,
        *,
        timeout: float,
        source_name: str | None = None,
    ) -> Any:
        """Open a bounded probe request using the central redirect rules."""

        return self._url_opener.open(
            request,
            timeout=timeout,
        )


class VideoFullProbeDownloader:
    """Download a bounded full video copy for metadata fallback probing."""

    def __init__(
        self,
        *,
        max_bytes: int,
        timeout_seconds: float,
        user_agent: str = DEFAULT_VIDEO_METADATA_USER_AGENT,
        transport: UrlopenTransport,
    ) -> None:
        self._max_bytes = max(1, int(max_bytes))
        self._timeout_seconds = max(0.1, float(timeout_seconds))
        self._user_agent = user_agent
        self._transport = transport

    @property
    def max_bytes(self) -> int:
        """Return the maximum temporary full-video probe size."""

        return self._max_bytes

    def download(self, *, url: str) -> Path | None:
        """Download the full response into a temporary file when bounded."""

        request = urllib.request.Request(
            url,
            headers={
                "Accept-Encoding": "identity",
                "User-Agent": self._user_agent,
            },
        )

        try:
            with self._transport.open(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                content_length = _optional_int(
                    response.headers.get("Content-Length")
                )
                if (
                    content_length is not None
                    and content_length > self._max_bytes
                ):
                    return None

                return self._write_response(response=response)
        except (
            OSError,
            TimeoutError,
            ValueError,
            urllib.error.URLError,
            RuntimeError,
        ):
            return None

    def _write_response(self, *, response: Any) -> Path | None:
        path: Path | None = None
        total = 0
        exceeded = False

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".video.full",
            ) as temp_file:
                path = Path(temp_file.name)
                while True:
                    chunk = response.read(1_048_576)
                    if not chunk:
                        break

                    total += len(chunk)
                    if total > self._max_bytes:
                        exceeded = True
                        break

                    temp_file.write(chunk)
        except (OSError, RuntimeError, ValueError):
            if path is not None:
                path.unlink(missing_ok=True)
            return None

        if exceeded or total <= 0:
            path.unlink(missing_ok=True)
            return None

        return path


class VideoTailProbeDownloader:
    """Download a bounded tail range for MP4 metadata fallback probing."""

    def __init__(
        self,
        *,
        tail_bytes: int,
        timeout_seconds: float,
        user_agent: str = DEFAULT_VIDEO_METADATA_USER_AGENT,
        transport: UrlopenTransport,
    ) -> None:
        self._tail_bytes = max(1, int(tail_bytes))
        self._timeout_seconds = max(0.1, float(timeout_seconds))
        self._user_agent = user_agent
        self._transport = transport

    def download(
        self,
        *,
        url: str,
        source_content_length: int | None,
    ) -> Path | None:
        """Download the last configured bytes of a remote video."""

        headers = {
            "Accept-Encoding": "identity",
            "User-Agent": self._user_agent,
        }

        if source_content_length is not None and source_content_length > 0:
            start = max(0, source_content_length - self._tail_bytes)
            end = max(0, source_content_length - 1)
            headers["Range"] = f"bytes={start}-{end}"
        else:
            headers["Range"] = f"bytes=-{self._tail_bytes}"

        request = urllib.request.Request(url, headers=headers)

        try:
            with self._transport.open(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                return self._write_response(response=response)
        except (
            OSError,
            TimeoutError,
            ValueError,
            urllib.error.URLError,
            RuntimeError,
        ):
            return None

    def _write_response(self, *, response: Any) -> Path | None:
        path: Path | None = None
        total = 0
        exceeded = False

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".video.tail",
            ) as temp_file:
                path = Path(temp_file.name)
                while True:
                    chunk = response.read(256_000)
                    if not chunk:
                        break

                    total += len(chunk)
                    if total > self._tail_bytes:
                        exceeded = True
                        break

                    temp_file.write(chunk)
        except (OSError, RuntimeError, ValueError):
            if path is not None:
                path.unlink(missing_ok=True)
            return None

        if exceeded or total <= 0:
            path.unlink(missing_ok=True)
            return None

        return path


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError, OverflowError):
        return None
